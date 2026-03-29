import {
  Header,
  HeaderContainer,
  HeaderName,
  HeaderMenuButton,
  HeaderGlobalBar,
  HeaderGlobalAction,
  SkipToContent,
  SideNav,
  SideNavItems,
  SideNavLink,
} from '@carbon/react';
import { Settings, UserAvatar } from '@carbon/icons-react';
import { useNavigate, useLocation } from 'react-router-dom';

const AppHeader = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <HeaderContainer
      render={({ isSideNavExpanded, onClickSideNavExpand }: { isSideNavExpanded: boolean; onClickSideNavExpand: () => void }) => (
        <>
          <Header aria-label="Alzheimer's MRI Analysis">
            <SkipToContent />
            <HeaderMenuButton
              aria-label="Open menu"
              onClick={onClickSideNavExpand}
              isActive={isSideNavExpanded}
            />
            <HeaderName href="#" prefix="AI" onClick={() => navigate('/')}>
              MRI Analysis
            </HeaderName>

            <HeaderGlobalBar>
              <HeaderGlobalAction aria-label="Settings" onClick={() => navigate('/settings')}>
                <Settings size={20} />
              </HeaderGlobalAction>
              <HeaderGlobalAction aria-label="User Avatar">
                <UserAvatar size={20} />
              </HeaderGlobalAction>
            </HeaderGlobalBar>
          </Header>

          <SideNav
            aria-label="Side navigation"
            expanded={isSideNavExpanded}
            isPersistent={false}
            onOverlayClick={onClickSideNavExpand}
          >
            <SideNavItems>
              <SideNavLink
                isActive={location.pathname === '/'}
                onClick={() => { navigate('/'); onClickSideNavExpand(); }}
              >
                Welcome
              </SideNavLink>
              <SideNavLink
                isActive={location.pathname === '/visualization'}
                onClick={() => { navigate('/visualization'); onClickSideNavExpand(); }}
              >
                Data Visualization
              </SideNavLink>
              <SideNavLink
                isActive={location.pathname === '/settings'}
                onClick={() => { navigate('/settings'); onClickSideNavExpand(); }}
              >
                Settings
              </SideNavLink>
            </SideNavItems>
          </SideNav>
        </>
      )}
    />
  );
};

export default AppHeader;
