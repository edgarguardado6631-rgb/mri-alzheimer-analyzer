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
import { Settings, UserAvatar, Home, Analytics, SettingsAdjust } from '@carbon/icons-react';
import { useNavigate, useLocation } from 'react-router-dom';

const AppHeader = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <HeaderContainer
      render={({ isSideNavExpanded, onClickSideNavExpand }: { isSideNavExpanded: boolean; onClickSideNavExpand: () => void }) => (
        <>
          <Header aria-label="NeuroScan AI — MRI Analysis">
            <SkipToContent />
            <HeaderMenuButton
              aria-label={isSideNavExpanded ? 'Close menu' : 'Open menu'}
              onClick={onClickSideNavExpand}
              isActive={isSideNavExpanded}
            />
            <HeaderName href="#" prefix="NeuroScan" onClick={(e) => { e.preventDefault(); navigate('/'); }}>
              AI
            </HeaderName>

            <HeaderGlobalBar>
              <HeaderGlobalAction aria-label="Settings" onClick={() => navigate('/settings')}>
                <Settings size={20} />
              </HeaderGlobalAction>
              <HeaderGlobalAction aria-label="User profile">
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
                renderIcon={Home}
                isActive={location.pathname === '/'}
                onClick={() => { navigate('/'); onClickSideNavExpand(); }}
              >
                Welcome
              </SideNavLink>
              <SideNavLink
                renderIcon={Analytics}
                isActive={location.pathname === '/visualization'}
                onClick={() => { navigate('/visualization'); onClickSideNavExpand(); }}
              >
                Data Visualization
              </SideNavLink>
              <SideNavLink
                renderIcon={SettingsAdjust}
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
